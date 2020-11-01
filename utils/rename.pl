#!/usr/bin/perl
'di';
'ig00';
#
# rename.pl 's|aaa|bbb|' *.txt
# 

if ($ARGV[0] eq '-i') {
    shift;
    if (open(TTYIN, "</dev/tty") && open(TTYOUT,">/dev/tty")) {
	$inspect++;
	select((select(TTYOUT),$|=1)[0]);
    } 
}
($op = shift) || die "Usage: rename [-i] perlexpr [filenames]\n";
if (!@ARGV) {
    @ARGV = <STDIN>;
    chop(@ARGV);
}
for (@ARGV) {
    unless (-e) {
	print STDERR "$0: $_: $!\n";
	$status = 1;
	next;
    } 
    $was = $_;
    eval $op;
    die $@ if $@;
    if ($was ne $_) {
	if ($inspect && -e) {
	    print TTYOUT "remove $_? ";
	    next unless <TTYIN> =~ /^y/i;
	} 
	unless (rename($was, $_)) {
	    print STDERR "$0: can't rename $was to $_: $!\n";
	    $status = 1;
	}
    } 
}
exit $status;
##############################################################################

	# These next few lines are legal in both Perl and nroff.

.00;			# finish .ig
 
'di			\" finish diversion--previous line must be blank
.nr nl 0-1		\" fake up transition to first page again
.nr % 0			\" start at page 1
';<<'.ex'; #__END__ ############# From here on it's a standard manual page ############
.TH RENAME 1 "July 30, 1990"
.AT 3
.SH NAME
rename \- renames multiple files
.SH SYNOPSIS
.B rename [-i] perlexpr [files]
.SH DESCRIPTION
.I Rename
renames the filenames supplied according to the rule specified as the
first argument.
The argument is a Perl expression which is expected to modify the $_
string in Perl for at least some of the filenames specified.
If a given filename is not modified by the expression, it will not be
renamed.
If no filenames are given on the command line, filenames will be read
via standard input.
.PP
The 
.B \-i
flag will prompt to remove the old file first if it exists.  This
flag will be ignored if there is no tty.
.PP
For example, to rename all files matching *.bak to strip the extension,
you might say
.nf

	rename 's/\e.bak$//' *.bak

.fi
To translate uppercase names to lower, you'd use
.nf

	rename 'y/A-Z/a-z/' *

.fi
To do the same thing but leave Makefiles unharmed:
.nf

	rename 'y/A-Z/a-z/ unless /^Make/' *

.fi
To rename all the *.f files to *.BAD, you'd use
.nf

	rename 's/\e.f$/.BAD/' *.f

.SH ENVIRONMENT
.fi
No environment variables are used.
.SH AUTHOR
Larry Wall
.SH "SEE ALSO"
mv(1)
.br
perl(1)
.SH DIAGNOSTICS
If you give an invalid Perl expression you'll get a syntax error.
.SH BUGS
.I Rename
does not check for the existence of target filenames unless you
specifty the 
.B \-i
flag, so use with care.
.ex

